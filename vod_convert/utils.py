import os


# ------------------------------------------------------------------------------
def matching_ids(
        annotations_dir: str,
        images_dir: str,
        annotations_ext: str,
        images_ext: str,
):
    """
    Given a directory and extension to use for image files and annotation files,
    find the matching file IDs across the two directories. Useful to find
    matching image and annotation files.

    For example, a match would be where we have two files
    <image_dir>/<file_id><images_ext> and <annotations_dir>/<file_id><annotations_ext>
    if <file_id> is the same for both files.

    :param annotations_dir:
    :param images_dir:
    :param annotations_ext:
    :param images_ext:
    :return:
    """

    # define a function to get all file IDs in a directory
    # where the file has the specified extension
    def file_ids(directory: str, extension: str):
        ids = []
        for file_name in os.listdir(directory):
            file_id, ext = os.path.splitext(file_name)
            if ext == extension:
                ids.append(file_id)
        return ids

    # get the list of file IDs matching to the relevant extensions
    ids_annotations = file_ids(annotations_dir, annotations_ext)
    ids_image = file_ids(images_dir, images_ext)

    # return the list of file IDs for the annotations and corresponding images
    return list(set(ids_annotations) & set(ids_image))
