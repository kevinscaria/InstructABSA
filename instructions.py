class InstructionsHandler:
    def __init__(self):
        self.ate = {}
        self.atsc = {}
        self.joint = {}

    def load_instruction_set1(self, ):
        self.ate['bos_instruct1'] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
                            Positive example 1-
                            input: I charge it at night and skip taking the cord with me because of the good battery life.
                            output: battery life
                            Positive example 2-
                            input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
                            output: features, iChat, Photobooth, garage band
                            Now complete the following example-
                            input: """
        
        self.ate['bos_instruct2'] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
                            Positive example 1-
                            input: With the great variety on the menu , I eat here often and never get bored.
                            output: menu
                            Positive example 2- 
                            input: Great food, good size menu, great service and an unpretensious setting.
                            output: food, menu, service, setting
                            Now complete the following example-
                            input: """

        self.ate['delim_instruct'] = ''
        self.ate['eos_instruct'] = ' \noutput:'

        self.atsc['bos_instruct'] = ''
        self.atsc['delim_instruct'] = ''
        self.atsc['eos_instruct'] = ''

        self.joint['bos_instruct'] = ''
        self.joint['delim_instruct'] = ''
        self.joint['eos_instruct'] = ''

    def load_instruction_set2(self, ):
        self.ate['bos_instruct1'] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
                            Positive example 1-
                            input: I charge it at night and skip taking the cord with me because of the good battery life.
                            output: battery life
                            Positive example 2-
                            input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
                            output: features, iChat, Photobooth, garage band
                            Negative example 1-
                            input: Speaking of the browser, it too has problems.
                            output: browser
                            Negative example 2-
                            input: The keyboard is too slick.
                            output: keyboard
                            Neutral example 1-
                            input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
                            output: battery
                            Neutral example 2-
                            input: Nightly my computer defrags itself and runs a virus scan.
                            output: virus scan
                            Now complete the following example-
                            input: """
        
        self.ate['bos_instruct2'] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
                            Positive example 1-
                            input: With the great variety on the menu , I eat here often and never get bored.
                            output: menu
                            Positive example 2- 
                            input: Great food, good size menu, great service and an unpretensious setting.
                            output: food, menu, service, setting
                            Negative example 1-
                            input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it.
                            output: toast, mayonnaise, bacon, ingredients, plate
                            Negative example 2-
                            input: The seats are uncomfortable if you are sitting against the wall on wooden benches.
                            output: seats
                            Neutral example 1-
                            input: I asked for seltzer with lime, no ice.
                            output: seltzer with lime
                            Neutral example 2-
                            input: They wouldnt even let me finish my glass of wine before offering another.
                            output: glass of wine
                            Now complete the following example-
                            input: """

        self.ate['delim_instruct'] = ''
        self.ate['eos_instruct'] = ' \noutput:'

        self.atsc['bos_instruct'] = ''
        self.atsc['delim_instruct'] = ''
        self.atsc['eos_instruct'] = ''

        self.joint['bos_instruct'] = ''
        self.joint['delim_instruct'] = ''
        self.joint['eos_instruct'] = ''