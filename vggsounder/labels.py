"""
VGGSounder Labels module for accessing video classification data.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class VideoData:
    """
    Container for video data including labels and metadata.

    Attributes:
        video_id: The unique identifier for the video
        labels: List of classification labels for the video
        meta_labels: Dictionary containing metadata like background_music, static_image, voice_over
        modalities: List of modalities for each label (A, AV, V)
    """

    video_id: str
    labels: List[str]
    meta_labels: Dict[str, bool]
    modalities: List[str]

    def __repr__(self):
        return f"VideoData(video_id='{self.video_id}', labels={len(self.labels)} items, meta_labels={self.meta_labels})"


class VGGSounder:
    """
    Main interface for accessing VGGSounder video classification data.

    Provides dict-like access to video data by video ID or index, with optional
    modality filtering.

    Example:
        >>> vggsounder = VGGSounder()
        >>> video_data = vggsounder["--U7joUcTCo_000000"]  # Access by video ID
        >>> video_data = vggsounder[0]  # Access by index
        >>> print(video_data.labels)
        >>> print(video_data.meta_labels)

        >>> # Filter by modality
        >>> vggsounder.set_modality("A")        # Audio labels (A + AV)
        >>> vggsounder.set_modality("A ONLY")   # Only pure audio labels (A only)
        >>> vggsounder.set_modality("V ONLY")   # Only pure visual labels (V only)
        >>> print(len(vggsounder))  # Returns count of videos with filtered labels

        >>> # Initialize with modality filter
        >>> audio_only = VGGSounder(modality="A ONLY")
    """

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        modality: Optional[str] = None,
    ):
        """
        Initialize the Labels object.

        Args:
            csv_path: Path to the CSV file. If None, looks for the default CSV file
                     in the package data directory.
            modality: Filter by modality:
                     - "A": audio (includes both "A" and "AV" labels)
                     - "V": visual (includes both "V" and "AV" labels)
                     - "AV": audio-visual (only "AV" labels)
                     - "A ONLY": audio only (only "A" labels, excludes "AV")
                     - "V ONLY": visual only (only "V" labels, excludes "AV")
                     - None: all modalities are included.
        """
        self.csv_path = self._resolve_csv_path(csv_path)
        self._current_modality: Optional[str] = modality

        self.__original_data: Dict[str, VideoData] = {}
        self._data: Dict[str, VideoData] = {}
        self._video_id_to_index: Dict[str, int] = {}
        self._index_to_video_id: Dict[int, str] = {}
        self._labels: List[str] = []

        self._load_data()
        self._load_labels()

    def _resolve_csv_path(self, csv_path: Optional[Union[str, Path]]) -> Path:
        """Resolve the path to the CSV file."""
        if csv_path is not None:
            return Path(csv_path)

        # Look for the CSV file in the package data directory first
        package_dir = Path(__file__).parent
        package_data_csv = package_dir / "data" / "vggsounder.csv"

        if package_data_csv.exists():
            return package_data_csv

        # Look for the CSV file in the project data directory (for development)
        project_data_csv = package_dir.parent / "data" / "vggsounder.csv"
        if project_data_csv.exists():
            return project_data_csv

        # If not found, look in common locations
        possible_paths = [
            Path("data/vggsounder.csv"),
            Path("../data/vggsounder.csv"),
            Path("vggsounder.csv"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Could not find vggsounder.csv. Please provide the path explicitly or "
            f"ensure the CSV file exists at one of these locations: {[str(p) for p in possible_paths + [str(package_data_csv), str(project_data_csv)]]}"
        )

    def _load_data(self):
        """Load data from the CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Dictionary to group rows by video_id
        video_groups: Dict[str, List[Dict[str, str]]] = {}

        with open(self.csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                video_id = row["video_id"]
                if video_id not in video_groups:
                    video_groups[video_id] = []
                video_groups[video_id].append(row)

        # Process each video group
        for video_id, rows in video_groups.items():
            labels = []
            modalities = []

            # Extract labels and modalities
            for row in rows:
                labels.append(row["label"])
                modalities.append(row["modality"])

            # Extract metadata from the first row (should be consistent across all rows for same video)
            first_row = rows[0]
            meta_labels = {
                "background_music": first_row["background_music"].lower() == "true",
                "static_image": first_row["static_image"].lower() == "true",
                "voice_over": first_row["voice_over"].lower() == "true",
            }

            # Create VideoData object
            video_data = VideoData(
                video_id=video_id,
                labels=labels,
                meta_labels=meta_labels,
                modalities=modalities,
            )

            # Store in original data (unfiltered)
            self.__original_data[video_id] = video_data

        # Apply modality filtering
        self._recompute_data()

    def _load_labels(self):
        """Load class order from classes.csv file."""
        # Look for classes.csv in the same directory as the main CSV
        # Look for the CSV file in the package data directory first
        package_dir = Path(__file__).parent
        classes_csv_path = package_dir / "data" / "classes.csv"

        if not classes_csv_path.exists():
            raise FileNotFoundError(f"classes.csv not found at {classes_csv_path}")
 
        # Read classes.csv and extract display_name column in order
        with open(classes_csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            self._labels = []

            for row in reader:
                display_name = row["display_name"]
                self._labels.append(display_name)

    def _recompute_data(self):
        """Recompute _data based on current modality filter."""
        self._data.clear()
        self._video_id_to_index.clear()
        self._index_to_video_id.clear()

        index = 0
        for video_id, video_data in self.__original_data.items():
            if self._current_modality is None:
                # No filtering, include all videos
                self._data[video_id] = video_data
                self._video_id_to_index[video_id] = index
                self._index_to_video_id[index] = video_id
                index += 1
            else:
                # Filter based on modality
                filtered_labels = []
                filtered_modalities = []

                for label, modality in zip(video_data.labels, video_data.modalities):
                    modality_upper = modality.upper()
                    current_upper = self._current_modality.upper()

                    # Check modality match based on filter type
                    if current_upper == "A ONLY":
                        match = modality_upper == "A"
                    elif current_upper == "V ONLY":
                        match = modality_upper == "V"
                    else:
                        # Original behavior for "A", "V", "AV"
                        match = current_upper in modality_upper

                    if match:
                        filtered_labels.append(label)
                        filtered_modalities.append(modality)

                # Only include video if it has labels with the specified modality
                if filtered_labels:
                    filtered_video_data = VideoData(
                        video_id=video_data.video_id,
                        labels=filtered_labels,
                        meta_labels=video_data.meta_labels,
                        modalities=filtered_modalities,
                    )
                    self._data[video_id] = filtered_video_data
                    self._video_id_to_index[video_id] = index
                    self._index_to_video_id[index] = video_id
                    index += 1

    def set_modality(self, modality: Optional[str] = None):
        """
        Set the modality filter for the dataset.

        Args:
            modality: Filter by modality:
                     - "A": audio (includes both "A" and "AV" labels)
                     - "V": visual (includes both "V" and "AV" labels)
                     - "AV": audio-visual (only "AV" labels)
                     - "A ONLY": audio only (only "A" labels, excludes "AV")
                     - "V ONLY": visual only (only "V" labels, excludes "AV")
                     - None: all modalities are included.
        """
        if modality is not None:
            modality = modality.upper()
            valid_modalities = {"A", "V", "AV", "A ONLY", "V ONLY"}
            if modality not in valid_modalities:
                raise ValueError(
                    f"Invalid modality '{modality}'. Must be one of {valid_modalities} or None"
                )

        self._current_modality = modality
        self._recompute_data()

    def get_modality(self) -> Optional[str]:
        """Get the current modality filter."""
        return self._current_modality

    def __getitem__(self, key: Union[str, int]) -> VideoData:
        """Get video data by video ID or index."""
        if isinstance(key, int):
            # Access by index
            if key < 0 or key >= len(self._data):
                raise IndexError(
                    f"Index {key} out of range for dataset of size {len(self._data)}"
                )
            video_id = self._index_to_video_id[key]
            return self._data[video_id]
        elif isinstance(key, str):
            # Access by video_id
            if key not in self._data:
                raise KeyError(f"Video ID '{key}' not found in dataset")
            return self._data[key]
        else:
            raise TypeError(
                f"Key must be str (video_id) or int (index), got {type(key)}"
            )

    def __contains__(self, video_id: str) -> bool:
        """Check if video ID exists in dataset."""
        return video_id in self._data

    def __len__(self) -> int:
        """Return number of unique videos in dataset."""
        return len(self._data)

    def __iter__(self):
        """Iterate over video IDs."""
        return iter(self._data.values())

    def keys(self):
        """Return video IDs."""
        return self._data.keys()

    def values(self):
        """Return VideoData objects."""
        return self._data.values()

    def items(self):
        """Return (video_id, VideoData) pairs."""
        return self._data.items()

    def get_videos_with_labels(self, *label_names: str) -> List[VideoData]:
        """
        Get all videos that contain any of the specified labels.

        Args:
            *label_names: Label names to search for

        Returns:
            List of VideoData objects containing any of the specified labels
        """
        result = []
        for video_data in self._data.values():
            if any(label in video_data.labels for label in label_names):
                result.append(video_data)
        return result

    def get_videos_with_meta(self, **meta_filters) -> List[VideoData]:
        """
        Get all videos that match the specified metadata filters.

        Args:
            **meta_filters: Metadata filters (e.g., background_music=True)

        Returns:
            List of VideoData objects matching the metadata filters
        """
        result = []
        for video_data in self._data.values():
            if all(
                video_data.meta_labels.get(key) == value
                for key, value in meta_filters.items()
            ):
                result.append(video_data)
        return result

    def get_all_labels(self) -> List[str]:
        """Get all unique labels in the dataset, ordered as in classes.csv."""
        return self._labels

    def stats(self) -> Dict[str, int]:
        """Get basic statistics about the dataset."""
        total_videos = len(self._data)
        total_labels = sum(len(video_data.labels) for video_data in self._data.values())
        unique_labels = len(self.get_all_labels())

        with_background_music = len(self.get_videos_with_meta(background_music=True))
        with_static_image = len(self.get_videos_with_meta(static_image=True))
        with_voice_over = len(self.get_videos_with_meta(voice_over=True))

        return {
            "total_videos": total_videos,
            "total_label_instances": total_labels,
            "unique_labels": unique_labels,
            "videos_with_background_music": with_background_music,
            "videos_with_static_image": with_static_image,
            "videos_with_voice_over": with_voice_over,
        }
