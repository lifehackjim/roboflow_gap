# roboflow_gap
Fill in the GAPs: Gather, Analyze, and Process computer vision images within seconds

## Introduction
There are three main steps in computer vision: gathering, analyzing, and processing images. 

The process of writing all of the tooling for using computer vision in python can be time-consuming and error-prone, and requires a lot of domain-specific knowledge, more than a little bit of trial and error, and at least some level of expertise in python.

This is where Roboflow GAP comes in. Roboflow GAP aims to be a python library that provides:
- Gather: A simple and easy-to-use interface for gathering images from various sources: camera on your computer, screenshot from your computer screen, movies or images from the web, movies or images from your computer, and more.
- Analyze: An easy way to use Roboflow's powerful computer vision libraries to analyze images and identify objects within them
- Process: An easy way to process images to add bounding boxes, labels, and other useful information. This would also allow for future lookup of information about the images, such as the wikipedia page for a given object, or other useful information.
- A CLI tool to make it easy to use Roboflow GAP from the command line
- Support for prompting the user for input when necessary, with self-help messages and error messages that are easy to understand
