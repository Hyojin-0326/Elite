#include <open3d/Open3D.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;
namespace fs = std::filesystem;

// ----------------------------
//            Scan
// ----------------------------
class Scan {
public:
    Eigen::Matrix4d pose;          // 4x4 homogeneous transform
    std::string path;              // path to .pcd file

    Scan(const Eigen::Matrix4d &pose_, const std::string &path_)
        : pose(pose_), path(path_), pcd_(nullptr) {}

    /**
     * Lazily‑loaded & transformed point cloud.
     * The shared_ptr ensures we do not copy heavy geometry data needlessly.
     */
    std::shared_ptr<open3d::geometry::PointCloud> get_pcd() const {
        if (!pcd_) {
            pcd_ = std::make_shared<open3d::geometry::PointCloud>();
            if (!open3d::io::ReadPointCloud(path, *pcd_)) {
                throw std::runtime_error("Failed to read point cloud: " + path);
            }
            pcd_->Transform(pose);
        }
        return pcd_;
    }

private:
    mutable std::shared_ptr<open3d::geometry::PointCloud> pcd_;
};

// ----------------------------
//        PointCloud
// ----------------------------
class PointCloud {
public:
    std::shared_ptr<open3d::geometry::PointCloud> cloud;

    PointCloud() : cloud(std::make_shared<open3d::geometry::PointCloud>()) {}
    explicit PointCloud(std::shared_ptr<open3d::geometry::PointCloud> c)
        : cloud(std::move(c)) {}

    // Down‑sample with a voxel grid
    PointCloud &downsample(double voxel_size = 0.2) {
        cloud = cloud->VoxelDownSample(voxel_size);
        return *this;
    }

    // Uniform color paint
    PointCloud &colorize(const std::vector<double> &color = {1.0, 0.0, 0.0}) {
        if (color.size() != 3)
            throw std::invalid_argument("color must have three elements [r,g,b]");
        cloud->PaintUniformColor({color[0], color[1], color[2]});
        return *this;
    }

    // Keep only points with z > height
    PointCloud &height_filter(double height = 0.5) {
        std::vector<size_t> idx;
        const auto &pts = cloud->points_;
        idx.reserve(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            if (pts[i](2) > height)
                idx.push_back(i);
        }
        cloud = cloud->SelectByIndex(idx);
        return *this;
    }

    // Quick visualization (debug)
    PointCloud &visualize() {
        auto axis = open3d::geometry::TriangleMesh::CreateCoordinateFrame(20.0, {0, 0, 0});
        open3d::visualization::DrawGeometries({cloud, axis});
        return *this;
    }

    // Save PCD with grayscale height‑based colors
    PointCloud &save(const std::string &out_path) {
        if (cloud->points_.empty())
            throw std::runtime_error("Point cloud is empty ‑ nothing to save.");

        // Compute min / max Z
        double min_z = std::numeric_limits<double>::max();
        double max_z = std::numeric_limits<double>::lowest();
        for (const auto &p : cloud->points_) {
            min_z = std::min(min_z, p(2));
            max_z = std::max(max_z, p(2));
        }
        double range = max_z - min_z + 1e-8;

        // Fill grayscale colors
        std::vector<Eigen::Vector3d> colors;
        colors.reserve(cloud->points_.size());
        for (const auto &p : cloud->points_) {
            double intensity = (p(2) - min_z) / range;
            colors.emplace_back(intensity, intensity, intensity);
        }
        cloud->colors_ = colors;

        if (!open3d::io::WritePointCloud(out_path, *cloud))
            throw std::runtime_error("Failed to write point cloud: " + out_path);
        return *this;
    }

    std::shared_ptr<open3d::geometry::PointCloud> get() const { return cloud; }
};

// ----------------------------
//           Session
// ----------------------------
class Session {
public:
    Session(const std::string &scans_dir, const std::string &pose_file)
        : scans_dir_(scans_dir), pose_file_(pose_file) {
        load_scans();
    }

    size_t size() const { return scans_.size(); }

    // Python‑style indexing access
    PointCloud get_item(size_t idx) const {
        if (idx >= scans_.size())
            throw std::out_of_range("index out of range");
        return PointCloud(scans_[idx].get_pcd());
    }

    Eigen::Matrix4d get_pose(size_t idx) const {
        if (idx >= scans_.size())
            throw std::out_of_range("index out of range");
        return scans_[idx].pose;
    }

    void update_pose(size_t idx, const Eigen::Matrix4d &T) {
        if (idx >= scans_.size())
            throw std::out_of_range("index out of range");
        scans_[idx].pose = T;
    }

    void save_pose(const std::string &pose_file) const {
        std::ofstream ofs(pose_file);
        if (!ofs)
            throw std::runtime_error("Cannot open file for writing: " + pose_file);

        for (const auto &s : scans_) {
            Eigen::Matrix<double, 3, 4> block = s.pose.block<3, 4>(0, 0);
            for (int i = 0; i < 12; ++i) {
                ofs << block(i / 4, i % 4);
                ofs << (i == 11 ? '\n' : ' ');
            }
        }
    }

    void save_pointcloud(size_t idx, const std::string &out_path, double voxel_size = 0.1) {
        get_item(idx).downsample(voxel_size).save(out_path);
    }

private:
    std::string scans_dir_;
    std::string pose_file_;
    std::vector<Scan> scans_;

    //-------------------- helpers --------------------//
    static std::string index_to_filename(const std::string &dir, size_t idx) {
        std::ostringstream oss;
        oss << std::setw(6) << std::setfill('0') << idx << ".pcd";
        return (fs::path(dir) / oss.str()).string();
    }

    std::vector<Eigen::Matrix4d> load_poses_from_file(const std::string &pose_file) {
        std::vector<Eigen::Matrix4d> poses;
        std::ifstream ifs(pose_file);
        if (!ifs)
            throw std::runtime_error("Cannot open pose file: " + pose_file);

        std::string line;
        while (std::getline(ifs, line)) {
            std::istringstream ss(line);
            std::vector<double> vals((std::istream_iterator<double>(ss)), std::istream_iterator<double>());
            if (vals.size() != 12)
                throw std::runtime_error("Invalid pose line: " + line);

            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    T(r, c) = vals[r * 4 + c];
            poses.push_back(T);
        }
        return poses;
    }

    void load_scans() {
        auto poses = load_poses_from_file(pose_file_);
        scans_.reserve(poses.size());

        for (size_t i = 0; i < poses.size(); ++i) {
            std::string pcd_path = index_to_filename(scans_dir_, i);
            if (!fs::exists(pcd_path))
                throw std::runtime_error("Missing scan file: " + pcd_path);
            scans_.emplace_back(poses[i], pcd_path);
        }
    }
};

// ----------------------------
//       PYBIND11 MODULE
// ----------------------------
PYBIND11_MODULE(session, m) {
    m.doc() = "pybind11 wrapper for Session and PointCloud";

    // PointCloud binding
    py::class_<PointCloud>(m, "PointCloud")
        .def("downsample", &PointCloud::downsample, py::arg("voxel_size") = 0.2, py::return_value_policy::reference_internal)
        .def("colorize", &PointCloud::colorize, py::arg("color") = std::vector<double>{1.0, 0.0, 0.0}, py::return_value_policy::reference_internal)
        .def("height_filter", &PointCloud::height_filter, py::arg("height") = 0.5, py::return_value_policy::reference_internal)
        .def("save", &PointCloud::save, py::arg("out_path"), py::return_value_policy::reference_internal)
        .def("get", &PointCloud::get, py::return_value_policy::move)
        .def("get_numpy", [](const PointCloud &pc) {
            return *(pc.get());  // Python에서 처리 가능한 포인트 클라우드로 반환
        }, py::return_value_policy::move);

    // Session binding
    py::class_<Session>(m, "Session")
        .def(py::init<const std::string &, const std::string &>(), py::arg("scans_dir"), py::arg("pose_file"))
        .def("__len__", &Session::size)
        .def("__getitem__", [](const Session &s, size_t idx) {
            return s.get_item(idx);
        }, py::return_value_policy::move)
        .def("__getitem__", [](const Session &s, py::slice slice) {
            py::ssize_t start, stop, step, slicelength;
            if (!slice.compute(s.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            auto combined = std::make_shared<open3d::geometry::PointCloud>();
            for (py::ssize_t i = start; i < stop; i += step)
                *combined += *s.get_item(static_cast<size_t>(i)).get();
            return PointCloud(combined);
        }, py::return_value_policy::move)
        .def("get_pose", &Session::get_pose, py::arg("idx"))
        .def("update_pose", &Session::update_pose, py::arg("idx"), py::arg("T"))
        .def("save_pose", &Session::save_pose, py::arg("pose_file"))
        .def("save_pointcloud", &Session::save_pointcloud, py::arg("idx"), py::arg("out_path"), py::arg("voxel_size") = 0.1);
}