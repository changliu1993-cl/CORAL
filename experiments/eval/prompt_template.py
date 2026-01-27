
class PromptTemplate:
    def __init__(self, prompt_dict, obj_token=None, ice_token=None):
        """
        Initialize the PromptTemplate class.
        
        :param prompt_dict: A dictionary where keys represent template types, and values represent template strings.
        :param obj_token: A dictionary mapping input fields (columns) to tokens within the templates.
        :param ice_token: A token used for additional processing (optional).
        """
        self.prompt_dict = prompt_dict
        self.obj_token = obj_token if obj_token is not None else {}
        self.ice_token = ice_token

    def generate_prompt(self, label, query, object_dict):
        """
        Generate a prompt based on the label, query, and input object_dict.
        
        :param label: The label that indicates which template to use from the prompt_dict.
        :param query: The query string to append at the end of the prompt.
        :param object_dict: A dictionary where keys are 'object', 'location', 'size', 'confidence'.
        :return: A generated prompt string with the data inserted into the template.
        """
        if label not in self.prompt_dict:
            raise ValueError(f"Label {label} not found in prompt dictionary.")
        
        # Get the appropriate template based on the label
        template = self.prompt_dict[label]

        # Generate the prompt based on object details
        object_details = ""
        if self.obj_token == '<OBJECT_DETAILS>':  # grounding_detail_dict format
            for i, obj in enumerate(object_dict, start=1):
                object_details += (
                    f"{i}. Object: {obj['object']}, Location: {obj['location']}, "
                    f"Size: {obj['size']}, Confidence: {obj['confidence']}%\n"
                )
            if object_details == "":
                prompt = query
            else:
                prompt = template.replace(self.obj_token , object_details)
                prompt = prompt.replace("<QUERY>", query.lower())
        
        elif self.obj_token == '<OBJECT_LIST>':  # grounding_dict format

            object_list_str = self.obj_ls2str(object_dict['objects'])
            if object_list_str == "":
                prompt = query
            else:
                # object_list_str = self.obj_ls2str([obj['object'] for obj in object_dict])
                prompt = template.replace(self.obj_token, object_list_str)
                prompt = prompt.replace("<QUERY>", query.lower())
        elif self.obj_token == '<OBJECT_DETECTED_A>':  
            # grounding_comb_dict format
            if len(object_dict['objects']) >= 2:
                obj1 = object_dict['objects'][0]
                obj2 = object_dict['objects'][1]
                prompt = template.replace(self.obj_token, "RAM Model (can detect many different objects, including rare or unusual ones) found:" + self.obj_ls2str(obj1))
                prompt = prompt.replace("<OBJECT_DETECTED_B>", "DETR Model (only detects common objects from a smaller list) found:"+self.obj_ls2str(obj2))
                prompt = prompt.replace("<QUERY>", query.lower())
            else:
                prompt = query
        else:
            raise ValueError(f"Object token {self.obj_token} not recognized.")    
        return prompt
    
    def obj_ls2str(self, obj_list):
        """
        Convert the list of objects to a simple string format.
        
        :param obj_list: A list of object names.
        :return: A string representation of the objects.
        """
        # if obj list is string, return the string
        if isinstance(obj_list, str):
            return obj_list.replace(" | ", ", ")     
        if len(obj_list) == 0:
            return ""
        if len(obj_list) == 1:
            return obj_list[0]
        elif len(obj_list) == 2:
            return obj_list[0] + " and " + obj_list[1]
        else:
            return ", ".join(obj_list[:-1]) + ", and " + obj_list[-1]


grounding_comb_dict = {
    0: (
        "List of detected objects in the image:\n<OBJECT_DETECTED_A>\n<OBJECT_DETECTED_B>\n"
        "Based on the detected objects above, <QUERY>"
    ),
    1: (
        "The most prominent objects detected are:\n<OBJECT_DETECTED_A>\n<OBJECT_DETECTED_B>\n"
        "Given these findings, <QUERY>"
    ),
    2: (
        "The following objects were detected in the image:\n<OBJECT_DETECTED_A>\n<OBJECT_DETECTED_B>\n"
        "With this information, <QUERY>"
    ),
    3: (
        "Here is a list of all objects detected in the image:\n<OBJECT_DETECTED_A>\n<OBJECT_DETECTED_B>\n"
        "Do not infer or hallucinate any additional objects. Using only the detected objects, <QUERY>"
    )
}

grounding_detail_dict = {
    0: (
        "List of detected objects in the image:\n<OBJECT_DETAILS>\n"
        "Based on the detected objects above, <QUERY>"
    ),
    1: (
        "The most prominent objects detected are:\n<OBJECT_DETAILS>\n"
        "Given these findings, <QUERY>"
    ),
    2: (
        "The following objects were detected in the image:\n<OBJECT_DETAILS>\n"
        "With this information, <QUERY>"
    ),
    3: (
        "Here is a list of all objects detected in the image:\n<OBJECT_DETAILS>\n"
        "Do not infer or hallucinate any additional objects. Using only the detected objects, <QUERY>"
    )
}

grounding_dict = {
    0: (
        "This image contains <OBJECT_LIST>. "
        "Based on this, <QUERY>"
    ),
    1: (
        "The image contains the following objects: <OBJECT_LIST>. "
        "Given these detected objects, <QUERY>"
    ),
    2: (
        "This image shows the following objects: <OBJECT_LIST>. "
        "Using this information, <QUERY>"
    ),
    3: (
        "The objects found in this image are: <OBJECT_LIST>. "
        "Considering this list of objects, <QUERY>"
    )
}

pope_grounding_dict = {
    0: (
        "This image contains only the following objects: <OBJECT_LIST>. "
        "Do not assume anything beyond these objects. Based solely on this list, <QUERY>"
    ),
    1: (
        "The detected objects in the image are: <OBJECT_LIST>. "
        "Answer based only on these objects. <QUERY>"
    ),
    2: (
        "This image shows the following objects: <OBJECT_LIST>. "
        "You must answer using only the objects in this list. Given these detected objects, <QUERY>"
    ),
    3: (
        "The objects found in this image are limited to: <OBJECT_LIST>. "
        "You should rely strictly on this list of objects and make no other guesses. Based on this, <QUERY>"
    )
}
