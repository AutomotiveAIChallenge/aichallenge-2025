#!/usr/bin/env python3
import argparse
from datetime import datetime
import os
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import matplotlib # Re-added import for matplotlib
from rosbag2_py import ConverterOptions
from rosbag2_py import SequentialReader
from rosbag2_py import StorageOptions

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

matplotlib.use("TkAgg")
plt.rcParams["font.size"] = 18
plt.style.use('dark_background')

def save_and_show_plot(folder_name, file_name, w_geom="800x600+10+10"):
    plt.get_current_fig_manager().window.wm_geometry(w_geom)
    save_dir = "result" + "/" + folder_name
    os.makedirs(save_dir, exist_ok=True) if not os.path.isdir(save_dir) else None
    [name, suffix] = file_name.split(".")
    plt.savefig(
        save_dir + "/" + name + "-" + datetime.now().strftime("%y-%m-%d-%H-%M-%S") + "." + suffix
    )
    plt.show()


def infer_configs(
    bag_uri: str,
    storage_ids={".db3": ("sqlite3", "cdr", "cdr"), ".mcap": ("mcap", "", "")},
) -> tuple:
    bag_uri_path = Path(bag_uri)
    if os.path.isfile(bag_uri):
        data_file = bag_uri_path
    else:
        data_file = next(p for p in bag_uri_path.glob("*") if p.suffix in storage_ids)
        if data_file.suffix not in storage_ids:
            raise ValueError(f"Unsupported storage id: {data_file.suffix}")
    return storage_ids[data_file.suffix]


def create_reader(input_uri: str) -> SequentialReader:
    """Create a reader object from the given input uri. The input uri could be a directory or a file."""
    storage_id, isf, osf = infer_configs(input_uri)
    storage_options = StorageOptions(
        uri=input_uri,
        storage_id=storage_id,
    )
    converter_options = ConverterOptions(
        input_serialization_format=isf,
        output_serialization_format=osf,
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def sync_topic(data1, data2) -> list:
    sync_data = []
    for idx1 in range(len(data1)):
        data = data2[0]
        for idx2 in range(len(data2)):
            if data1[idx1][0] < data2[idx2][0]:
                break
            data = data2[idx2]
        sync_data.append(data)
    return sync_data



analyze_topic_list = [
    "/localization/kinematic_state",
    "/localization/acceleration",
]


class Analyzer:
    def __init__(self, input_dir, output_dir):
        self.input_bag_dir = input_dir
        self.folder_name = output_dir
        self.window_size = 10
        self.file_name = "planning_performance.png"

    def _read_bag_data(self):
        reader = create_reader(self.input_bag_dir)
        pose_time_stamp = []
        pose_speed = []
        pose_acceleration = []
        topic_type_list = {}

        for topic_type in reader.get_all_topics_and_types():
            topic_type_list[topic_type.name] = topic_type.type

        while reader.has_next():
            topic_name, msg, stamp = reader.read_next()
            stamp = stamp * 1e-9
            if topic_name in analyze_topic_list:
                data = deserialize_message(msg, get_message(topic_type_list[topic_name]))
                if topic_name == "/localization/kinematic_state":
                    if data.pose.pose.position.x != 0.0 or data.pose.pose.position.y != 0.0:
                        pose_time_stamp.append(
                            [stamp, data.pose.pose.position.x, data.pose.pose.position.y]
                        )
                        pose_speed.append([stamp, data.twist.twist.linear.x])
                elif topic_name == "/localization/acceleration":
                    pose_acceleration.append([stamp, data.accel.accel.linear.x])
        
        return pose_time_stamp, pose_speed, pose_acceleration, topic_type_list

    def _sync_and_filter_data(self, pose_time_stamp, pose_speed, pose_acceleration):
        if pose_speed and pose_time_stamp:
            pose_speed_filter = sync_topic(pose_time_stamp, pose_speed)
        else:
            pose_speed_filter = []

        if pose_acceleration and pose_time_stamp:
            pose_acceleration_filter = sync_topic(pose_time_stamp, pose_acceleration)
        else:
            pose_acceleration_filter = []

        return pose_speed_filter, pose_acceleration_filter

    def _plot_data(self, ax, pose_x, pose_y, values, title, cbar_label, cmap):
        if values:
            plot_cm = ax.scatter(pose_x[:len(values)], pose_y[:len(values)], 
                                 c=values, cmap=cmap, s=3.0, label=title)
            ax.set_title(title, fontsize=22)
            cbar = plt.colorbar(plot_cm, ax=ax, shrink=0.9, aspect=15, pad=0.05)
            cbar.set_label(cbar_label, fontsize=18)
            cbar.ax.tick_params(labelsize=16)
        else:
            ax.text(0.5, 0.5, f"No {title} Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)

    def _create_plots(self, pose_time_stamp, pose_speed_filter, pose_acceleration_filter):
        pose_x = [d[1] for d in pose_time_stamp]
        pose_y = [d[2] for d in pose_time_stamp]

        # Calculate figure size based on aspect ratio
        width = 16
        height = width * 0.6
        fig = plt.figure(figsize=(width, height))
        
        # Create subplots with adjusted spacing
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.4)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Set axis labels and grid
        ax1.set_xlabel("x [m]", fontsize=18)
        ax1.set_ylabel("y [m]", fontsize=18)
        ax1.grid(True, color='white', linestyle='-', alpha=0.3)
        ax1.set_aspect('equal')
        ax1.tick_params(axis='both', which='major', labelsize=16)
        
        ax2.set_xlabel("x [m]", fontsize=18)
        ax2.set_ylabel("y [m]", fontsize=18)
        ax2.grid(True, color='white', linestyle='-', alpha=0.3)
        ax2.set_aspect('equal')
        ax2.tick_params(axis='both', which='major', labelsize=16)

        if pose_time_stamp:
            min_x, max_x = min(pose_x), max(pose_x)
            min_y, max_y = min(pose_y), max(pose_y)

            buffer_x = (max_x - min_x) * 0.1 if (max_x - min_x) > 0 else 1.0
            buffer_y = (max_y - min_y) * 0.1 if (max_y - min_y) > 0 else 1.0

            ax1.set_xlim(min_x - buffer_x, max_x + buffer_x)
            ax1.set_ylim(min_y - buffer_y, max_y + buffer_y)
            ax2.set_xlim(min_x - buffer_x, max_x + buffer_x)
            ax2.set_ylim(min_y - buffer_y, max_y + buffer_y)
            
        self._plot_data(ax1, pose_x, pose_y, [d[1] for d in pose_speed_filter], "Velocity", 'Velocity [m/s]', cm.jet)
        self._plot_data(ax2, pose_x, pose_y, [d[1] for d in pose_acceleration_filter], "Acceleration", 'Acceleration [m/s^2]', cm.jet)

        # Adjust layout with specific padding
        plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85, wspace=0.2)
        return fig, ax1, ax2

    def plot(self):
        pose_time_stamp, pose_speed, pose_acceleration, topic_type_list = self._read_bag_data()
        pose_speed_filter, pose_acceleration_filter = self._sync_and_filter_data(pose_time_stamp, pose_speed, pose_acceleration)
        
        # Now call the new method to create plots
        fig, ax1, ax2 = self._create_plots(pose_time_stamp, pose_speed_filter, pose_acceleration_filter)

        save_and_show_plot(self.folder_name, self.file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output", default=datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
    args = parser.parse_args()

    analyzer = Analyzer(args.input, args.output)
    analyzer.plot()


if __name__ == "__main__":
    main()
