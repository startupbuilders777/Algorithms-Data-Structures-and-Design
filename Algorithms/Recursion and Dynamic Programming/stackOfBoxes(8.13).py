# NOT DONE

'''
Boxes have width, height, depth
boxes cant be rotated and can only be stacked on top of one another if each box in the stack is strictly 
larger than the box above it in width, height, and depth. Impleemnt a method to compute the heights of the tallest possible
stack. the hieght of s stack is the sume of the heights of each box
'''


boxes = [ [3,5,6], [3,4,4], [1,1,1], [1,1,10], [4,3,2] ]


def stack_of_boxes(boxes):
    #index 0 is w
    #index 1 is d
    # index 3 is h

    # have to order boxes in the way they can be stacked on top of each other
    # process a box, either its in the stack or not in the stack, and compute the max of those 2 decisions. 
    # 

    # sort from largest to smallest, with large meaning a box has greater 3 dimensions then the other box.
    # 2 boxes are considered the "same" when a box has some dimensions larger than that box and some dimensions smaller than that box. 
