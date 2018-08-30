

#SHould sort string so you can removeCharacter in log n
#ONly works with strings with unique characters for now

def getAllPerms(str):

    str.sort()  #O(nlogn)

    def getAllPermsRecur(str):
        allPerms = []
        for i in range(0,len(str)):
            character = str[i]
            lstOfAllPerms = getAllPerms(removeCharacter(c, str))

            for j in range(0, len(lstOfAllPerms)):
                allPerms.append(i + lstOfAllPerms[j])

        return allPerms;

    return getAllPermsRecur(str)


def removeCharacter(c, str, l1, r1, l2, r2):  # Use Binary Search
    length = len(str)
    mid = int(length / 2)
    print(length)
    print(mid)


    if (length == 0):
        return
    if (str[mid] == c):
        return str[0:int(mid)] + str[int(mid + 1):length]
    elif (str[mid] < c):
        return str[0:int(mid)] + removeCharacter(c, str[int(mid): length])
    else:
        return removeCharacter(c, str[0: int(mid)]) + str[int(mid): length]


print(removeCharacter("A", "BVASSFEW"))
print(removeCharacter("B", "BVASSFEW"))
print(removeCharacter("S", "BVASSFEW"))
print(removeCharacter("W", "BVASSFEW"))
print(removeCharacter("F", "BVASSFEW"))
