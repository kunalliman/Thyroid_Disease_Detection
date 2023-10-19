from datetime import datetime


class App_Logger:
    def __init__(self):
        pass

    def log(self, file_object, log_message):         # Will write logs when in the provided file_object which will be declared at starting of the code for each module                                              
        self.now = datetime.now()                    # and then the messages may keep changing for each step in the module  
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message +"\n")
