"""
This module provides the Participant class, each instance representing one
participant or collection folder under ``participants/``.
"""

from pelican_nlp.utils.lpds_paths import unit_id_from_folder, unit_kind


class Participant:
    def __init__(self, name):
        self.name = name
        self.kind = unit_kind(name)
        self.participantID = unit_id_from_folder(name)
        self.documents = []

    def __repr__(self):
        return f"Participant(name={self.name}, kind={self.kind})"
