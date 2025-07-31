import os
import sys
import yaml

from alignment.map_zipper import MapZipper
from dynamic_removal.map_remover import MapRemover
from map_update.map_updater import MapUpdater

from utils.session import Session
from utils.session_map import SessionMap
from utils.logger import logger


class ELite:
    def __init__(
        self,
        config_path: str
    ):
        self.map_zipper = MapZipper(config_path)
        self.map_remover = MapRemover(config_path)
        self.map_updater = MapUpdater(config_path)

        # read config_path yaml
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

    def run_elite(self):
        ###
        logger.info("Starting ELite...")

        # 1) Map Zipper
        logger.info("Running Map Zipper...")
        self.map_zipper.load_source_session()
        if "prev_output_dir" in self.params["settings"] and\
              os.path.exists(self.params["settings"]["prev_output_dir"]):
            self.map_zipper.load_target_session_map()
        aligned_session = self.map_zipper.run()
        logger.info("Map Zipper finished.")

        # 2) Map Remover
        logger.info("Running Map Remover...")
        self.map_remover.load(aligned_session)
        cleaned_session_map = self.map_remover.run()
        logger.info("Map Remover finished.")

        # 3) Map Updater
        logger.info("Running Map Updater...")
        prev_lifelong_map = SessionMap()
        if "prev_output_dir" in self.params["settings"] and\
              os.path.exists(self.params["settings"]["prev_output_dir"]):
            prev_lifelong_map.load(self.params["settings"]["prev_output_dir"], is_global=True)
            self.map_updater.load(prev_lifelong_map, cleaned_session_map)
            self.map_updater.run()
        else:
            self.map_updater.save(cleaned_session_map)
        logger.info("Map Updater finished.")

        ###
        logger.info("ELite finished.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        sys.exit("Need config file path: python run_elite.py <config_path>")

    elite = ELite(config_path)
    elite.run_elite()