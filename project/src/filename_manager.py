
class fname_manager:
    """
    this class manages the file names for all saving files
    """

    def __init__(self):
        """
        constructor
        """
        self.harris_save_file_last_letter = ord('b')
        self.matching_save_file_last_letter = ord('b')
        self.RANSAC_matching_save_file_last_letter = ord('b')
        self.image_stitch_save_file_last_letter = ord('b')

    def get_2_harris_output_filenames(self):
        """
        :return: return 2 file names for harris corner output
        """
        # set the filenames
        harris_save_file_name_1 = '1' + chr(self.harris_save_file_last_letter) + '.png'
        self.harris_save_file_last_letter += 1

        harris_save_file_name_2 = '1' + chr(self.harris_save_file_last_letter) + '.png'
        self.harris_save_file_last_letter += 1

        return (harris_save_file_name_1, harris_save_file_name_2)

    def get_matching_output_filename(self):
        """
        :return: return 1 file name for feature matching
        """
        # set the filename
        matching_save_file_name = '2' + chr(self.matching_save_file_last_letter) + '.png'
        self.matching_save_file_last_letter += 1

        return matching_save_file_name

    def get_RANSAC_matching_output_filename(self):
        """
        :return: return 1 file name for matching image after RANSAC
        """
        # set the filename
        RANSAC_matching_save_file_name = '3' + chr(self.RANSAC_matching_save_file_last_letter) + '.png'
        self.RANSAC_matching_save_file_last_letter += 1

        return RANSAC_matching_save_file_name

    def get_image_stitch_output_filename(self):
        """
        :return: return 1 file name for image stitching
        """
        # set the filename
        RANSAC_matching_save_file_name = '4' + chr(self.image_stitch_save_file_last_letter) + '.png'
        self.image_stitch_save_file_last_letter += 1

        return RANSAC_matching_save_file_name

