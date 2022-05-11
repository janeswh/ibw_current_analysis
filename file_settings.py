"""File settings for running multiple analyses across datasets"""
from enum import Enum


class FileSettings(object):

    DATA_FOLDER = "/home/jhuang/Documents/phd_projects/injected_GC_data/data"
    IGNORED = {"esc_unusable"}
    TABLES_FOLDER = (
        "/home/jhuang/Documents/phd_projects/injected_GC_data/tables"
    )
    FIGURES_FOLDER = (
        "/home/jhuang/Documents/phd_projects/injected_GC_data/figures"
    )
    PAPER_FIGURES_FOLDER = (
        "/home/jhuang/Documents/phd_projects/injected_GC_data/paper_figures"
    )
