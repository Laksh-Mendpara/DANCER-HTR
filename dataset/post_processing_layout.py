import numpy as np
from Datasets.dataset_formatters.read2016_formatter import SEM_MATCHING_TOKENS as READ_MATCHING_TOKENS


class PostProcessingModule:
    """
    Forward pass post processing
    Add/remove layout tokens only to:
     - respect token hierarchy
     - complete/remove unpaired tokens
    """

    def __init__(self):
        self.prediction = None
        self.confidence = None
        self.num_op = 0

    def post_processing(self):
        raise NotImplementedError

    def post_process(self, prediction, confidence_score=None):
        """
        Apply dataset-specific post-processing
        """
        self.prediction = list(prediction)
        self.confidence = list(confidence_score) if confidence_score is not None else None
        if self.confidence is not None:
            assert len(self.prediction) == len(self.confidence)
        return self.post_processing()

    def insert_label(self, index, label):
        """
        Insert token at specific index. The associated confidence score is set to 0.
        """
        self.prediction.insert(index, label)
        if self.confidence is not None:
            self.confidence.insert(index, 0)
        self.num_op += 1

    def del_label(self, index):
        """
        Remove the token at a specific index.
        """
        del self.prediction[index]
        if self.confidence is not None:
            del self.confidence[index]
        self.num_op += 1


class PostProcessingModuleREAD(PostProcessingModule):
    """
    Specific post-processing for the READ 2016 dataset at single-page and double-page levels
    """
    def __init__(self):
        super(PostProcessingModuleREAD, self).__init__()

        self.matching_tokens = READ_MATCHING_TOKENS
        self.reverse_matching_tokens = dict()
        for key in self.matching_tokens:
            self.reverse_matching_tokens[self.matching_tokens[key]] = key

    def post_processing_page_labels(self):
        """
        Correct tokens of page detection.
        """
        ind = 0
        while ind != len(self.prediction):
            # Label must start with a begin-page token
            if ind == 0 and self.prediction[ind] != "ⓟ":
                self.insert_label(0, "ⓟ")
                continue
            # There cannot be tokens out of begin-page end-page scope: begin-page must be preceded by end-page
            if self.prediction[ind] == "ⓟ" and ind != 0 and self.prediction[ind - 1] != "Ⓟ":
                self.insert_label(ind, "Ⓟ")
                continue
            # There cannot be tokens out of begin-page end-page scope: end-page must be followed by begin-page
            if self.prediction[ind] == "Ⓟ" and ind < len(self.prediction) - 1 and self.prediction[ind + 1] != "ⓟ":
                self.insert_label(ind + 1, "ⓟ")
            ind += 1
        # Label must start with a begin-page token even for empty prediction
        if len(self.prediction) == 0:
            self.insert_label(0, "ⓟ")
            ind += 1
        # Label must end with a end-page token
        if self.prediction[-1] != "Ⓟ":
            self.insert_label(ind, "Ⓟ")

    def post_processing(self):
        """
        Correct tokens of page number, section, body and annotations.
        """
        self.post_processing_page_labels()
        ind = 0
        begin_token = None
        in_section = False
        while ind != len(self.prediction):
            # each tags must be closed while changing page
            if self.prediction[ind] == "Ⓟ":
                if begin_token is not None:
                    self.insert_label(ind, self.matching_tokens[begin_token])
                    begin_token = None
                    ind += 1
                elif in_section:
                    self.insert_label(ind, self.matching_tokens["ⓢ"])
                    in_section = False
                    ind += 1
                else:
                    ind += 1
                continue
            # End token is removed if the previous begin token does not match with it
            if self.prediction[ind] in "ⓃⒶⒷ":
                if begin_token == self.reverse_matching_tokens[self.prediction[ind]]:
                    begin_token = None
                    ind += 1
                else:
                    self.del_label(ind)
                continue
            if self.prediction[ind] == "Ⓢ":
                # each sub-tags must be closed while closing section
                if in_section:
                    if begin_token is None:
                        in_section = False
                        ind += 1
                    else:
                        self.insert_label(ind, self.matching_tokens[begin_token])
                        begin_token = None
                        ind += 2
                else:
                    self.del_label(ind)
                continue
            if self.prediction[ind] == "ⓢ":
                # A sub-tag must be closed before opening a section
                if begin_token is not None:
                    self.insert_label(ind, self.matching_tokens[begin_token])
                    begin_token = None
                    ind += 1
                # A section must be closed before opening a new one
                elif in_section:
                    self.insert_label(ind, "Ⓢ")
                    in_section = False
                    ind += 1
                else:
                    in_section = True
                    ind += 1
                continue
            if self.prediction[ind] == "ⓝ":
                # Page number cannot be in section: a started section must be closed
                if begin_token is None:
                    if in_section:
                        in_section = False
                        self.insert_label(ind, "Ⓢ")
                        ind += 1
                    begin_token = self.prediction[ind]
                    ind += 1
                else:
                    self.insert_label(ind, self.matching_tokens[begin_token])
                    begin_token = None
                    ind += 1
                continue
            if self.prediction[ind] in "ⓐⓑ":
                # Annotation and body must be in section
                if begin_token is None:
                    if in_section:
                        begin_token = self.prediction[ind]
                        ind += 1
                    else:
                        in_section = True
                        self.insert_label(ind, "ⓢ")
                        ind += 1
                # Previous sub-tag must be closed
                else:
                    self.insert_label(ind, self.matching_tokens[begin_token])
                    begin_token = None
                    ind += 1
                continue
            ind += 1
        res = "".join(self.prediction)
        if self.confidence is not None:
            return res, np.array(self.confidence)
        return res
