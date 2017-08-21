def hi():
    def hiRecursive():
        print("hi")
        hiRecursive()
    hiRecursive()

hi()