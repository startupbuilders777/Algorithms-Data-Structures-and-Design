#!/usr/bin/python

class Car:
    'THIS IS SOME DOCUMENTATION FOR CARS'
    carCount = 0

    def __init__(self, name, manufacturer, licensePlate, size, speed):
        self.name = name
        self.manufacturer = manufacturer
        self.speed = speed
        self.licensePlate = licensePlate
        self.size = size
        Car.carCount += 1

    def displayLicensePlate(self):
        print("License Plate is %s" % self.licensePlate)

    def displayName(self):
        print("Manufacturer : ", self.manufacturer, ", Name: ", self.name)

#INITIIAING THE CARS
carA = Car("fook", "TSLA", "L4T2K1", 5, 120);
carB = Car("fook2", "FORD", "l4t3p3", 2, 200);

carA.displayLicensePlate()
carB.displayName()
#print("Total Cars %d", Car.carCount)


#You can add, remove, or modify attributes of classes and objects at any time −

carA.age = 8 #Add an age attribute
carA.age = 9 #Modify Attr
del carA.age #Delete "age" attribute

hasattr(carA, "age") #Returns true if "age" attribute exists
getattr(carA, "age") #Returns the value of attribute
setattr(carA, "age", 69)
delattr(carA, "age") #Deletes

'''
Built-In Class Attributes
Every Python class keeps following built-in attributes and they can be accessed using dot operator like any other attribute −

__dict__: Dictionary containing the class's namespace.

__doc__: Class documentation string or none, if undefined.

__name__: Class name.

__module__: Module name in which the class is defined. This attribute is "__main__" in interactive mode.

__bases__: A possibly empty tuple containing the base classes, in the order of their occurrence in the base class list.

'''

print(Car.__doc__)
print(Car.__name__)
print(Car.__module__)
print(Car.__bases__)
print(Car.__dict__)



