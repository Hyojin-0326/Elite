#pragma once

#include <open3d/Open3D.h>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>

namespace elite {

class Scan {
public:
    Scan(const Eigen::Matrix4d& pose, const std::string& path);

    const open3d::geometry::PointCloud& GetPointCloud();
    const Eigen::Matrix4d& GetPose() const;

private:
    Eigen::Matrix4d pose_;
    std::string path_;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_;
};


class PointCloud {
public:
    PointCloud(const open3d::geometry::PointCloud& cloud);

    PointCloud& Downsample(double voxel_size = 0.2);
    PointCloud& Colorize(const Eigen::Vector3d& color = {1.0, 0.0, 0.0});
    PointCloud& HeightFilter(double height = 0.5);
    void Visualize() const;
    void Save(const std::string& path) const;
    open3d::geometry::PointCloud Get() const;

private:
    open3d::geometry::PointCloud cloud_;
};


class Session {
public:
    Session(const std::string& scans_dir, const std::string& pose_file);

    size_t Size() const;
    PointCloud GetPointCloud(int idx) const;
    PointCloud GetPointCloudRange(int start, int end) const;

    const Eigen::Matrix4d& GetPose(int idx) const;
    void UpdatePose(int idx, const Eigen::Matrix4d& pose);
    void SavePose(const std::string& pose_file) const;
    void SavePointCloud(int idx, const std::string& path, double voxel_size = 0.1) const;

private:
    std::vector<Scan> scans_;
    std::vector<Eigen::Matrix4d> LoadPoses(const std::string& pose_file) const;
};

} // namespace elite
