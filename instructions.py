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

        self.atsc['bos_instruct1'] = """Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
                            Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
                            Positive example 1-
                            input: I charge it at night and skip taking the cord with me because of the good battery life. The aspect is battery life.
                            output: positive
                            Positive example 2-
                            input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!. The aspect is garage band.
                            output: positive
                            Now complete the following example-
                            input: """
        
        self.atsc['bos_instruct1'] = """Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
                            Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
                            Positive example 1-
                            input: With the great variety on the menu , I eat here often and never get bored. The aspect is menu.
                            output: positive
                            Positive example 2- 
                            input: Great food, good size menu, great service and an unpretensious setting. The aspect is food.
                            output: positive
                            Now complete the following example-
                            input: """
        self.atsc['delim_instruct'] = ''
        self.atsc['eos_instruct'] = ''

        self.joint['bos_instruct1'] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
                            Positive example 1-
                            input: I charge it at night and skip taking the cord with me because of the good battery life.
                            output: battery life:positive, 
                            Positive example 2-
                            input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
                            output: features:positive, iChat:positive, Photobooth:positive, garage band:positive
                            Now complete the following example-
                            input: """
        
        self.joint['bos_instruct2'] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
                            Positive example 1-
                            input: With the great variety on the menu , I eat here often and never get bored.
                            output: menu:positive
                            Positive example 2- 
                            input: Great food, good size menu, great service and an unpretensious setting.
                            output: food:positive, menu:positive, service:positive, setting:positive
                            Now complete the following example-
                            input: """
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

        self.atsc['bos_instruct1'] = """Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
                            Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
                            Positive example 1-
                            input: I charge it at night and skip taking the cord with me because of the good battery life. The aspect is battery life.
                            output: positive
                            Positive example 2-
                            input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!. The aspect is garage band.
                            output: positive
                            Negative example 1-
                            input: Speaking of the browser, it too has problems. The aspect is browser.
                            output: negative
                            Negative example 2-
                            input: The keyboard is too slick. The aspect is keyboard.
                            output: negative
                            Neutral example 1-
                            input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset. The aspect is battery.
                            output: neutral
                            Neutral example 2-
                            input: Nightly my computer defrags itself and runs a virus scan. The aspect is virus scan.
                            output: neutral
                            Now complete the following example-
                            input: """
        
        self.atsc['bos_instruct1'] = """Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
                            Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
                            Positive example 1-
                            input: With the great variety on the menu , I eat here often and never get bored. The aspect is menu.
                            output: positive
                            Positive example 2- 
                            input: Great food, good size menu, great service and an unpretensious setting. The aspect is food.
                            output: positive
                            Negative example 1-
                            input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it. The aspect is toast.
                            output: negative
                            Negative example 2-
                            input: The seats are uncomfortable if you are sitting against the wall on wooden benches. The aspect is seats.
                            output: negative
                            Neutral example 1-
                            input: I asked for seltzer with lime, no ice. The aspect is seltzer with lime.
                            output: neutral
                            Neutral example 2-
                            input: They wouldnt even let me finish my glass of wine before offering another. The aspect is glass of wine.
                            output: neutral
                            Now complete the following example-
                            input: """
        self.atsc['delim_instruct'] = ' The aspect is '
        self.atsc['eos_instruct'] = '.\noutput:'

        self.joint['bos_instruct1'] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
                            Positive example 1-
                            input: I charge it at night and skip taking the cord with me because of the good battery life.
                            output: battery life:positive, 
                            Positive example 2-
                            input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
                            output: features:positive, iChat:positive, Photobooth:positive, garage band:positive
                            Negative example 1-
                            input: Speaking of the browser, it too has problems.
                            output: browser:negative
                            Negative example 2-
                            input: The keyboard is too slick.
                            output: keyboard:negative
                            Neutral example 1-
                            input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
                            output: battery:neutral
                            Neutral example 2-
                            input: Nightly my computer defrags itself and runs a virus scan.
                            output: virus scan:neutral
                            Now complete the following example-
                            input: """
        
        self.joint['bos_instruct2'] = """Definition: The output will be the aspects (both implicit and explicit) and the aspects sentiment polarity. In cases where there are no aspects the output should be noaspectterm:none.
                            Positive example 1-
                            input: With the great variety on the menu , I eat here often and never get bored.
                            output: menu:positive
                            Positive example 2- 
                            input: Great food, good size menu, great service and an unpretensious setting.
                            output: food:positive, menu:positive, service:positive, setting:positive
                            Negative example 1-
                            input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it.
                            output: toast:negative, mayonnaise:negative, bacon:negative, ingredients:negative, plate:negative
                            Negative example 2-
                            input: The seats are uncomfortable if you are sitting against the wall on wooden benches.
                            output: seats:negative
                            Neutral example 1-
                            input: I asked for seltzer with lime, no ice.
                            output: seltzer with lime:neutral
                            Neutral example 2-
                            input: They wouldnt even let me finish my glass of wine before offering another.
                            output: glass of wine:neutral
                            Now complete the following example-
                            input: """
        self.joint['delim_instruct'] = ''
        self.joint['eos_instruct'] = ' \noutput:'