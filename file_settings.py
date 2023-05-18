"""File settings for running multiple analyses across datasets"""
from enum import Enum


class FileSettings(object):

    DATA_FOLDER = "/home/jhuang/Documents/phd_projects/injected_GC_data/data"
    IGNORED = {"p2_5wpi", "p2_4wpi", "p2_6wpi", "extra_sweeps"}
    TABLES_FOLDER = (
        "/home/jhuang/Documents/phd_projects/injected_GC_data/tables"
    )
    FIGURES_FOLDER = (
        "/home/jhuang/Documents/phd_projects/injected_GC_data/figures"
    )
    PAPER_FIGURES_FOLDER = (
        "/home/jhuang/Documents/phd_projects/injected_GC_data/paper_figures"
    )

    PRESENTATION_FIGURES_FOLDER = "/home/jhuang/Documents/phd_projects/injected_GC_data/presentation_figures"

    PAPER_FIGS_DATA_FOLDER = (
        "/home/jhuang/Documents/phd_projects/injected_GC_data/tables/"
        "paper_figs_data"
    )
