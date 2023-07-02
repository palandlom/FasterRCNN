class DatasetSample:

    def __init__(self, sample_id,  pict_path="", objects="", annot_path=""):
        self.id = sample_id
        self.pict_path = pict_path
        self.annot_path = annot_path
        self.objects = objects

# class InLabel:
