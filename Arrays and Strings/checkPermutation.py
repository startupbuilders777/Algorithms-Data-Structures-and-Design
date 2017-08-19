'''Given 2 strings, write a method to decide if one is a permutation of the other'''

def checkPermutation(str1, str2):
    count1 = {}
    count2 = {}

    if(len(str1) != len(str2)):
        return False

    for i in str1:
        if(count1.get(i) is None):
            count1[i] = 1
        else:
            count1[i] += 1

    for i in str2:
        if(count2.get(i) is None):
            count2[i] = 1
        else:
            count2[i] += 1

    for letter, count in count1.items():
         if(count2[letter] != count):
             return False
    return True

print(checkPermutation("hi", "ih"))
print(checkPermutation("car", "racx"))
print(checkPermutation("car", "rac"))

