

#Getting all the permutations of a string:

#UNIQUE STRINGS

'''

str = "this is string example....wow!!! this is really string"
print str.replace("is", "was")
print str.replace("is", "was", 3)

thwas was string example....wow!!! thwas was really string
thwas was string example....wow!!! thwas is really string

Cool string fuctions

'''

def permutations(string):
    if len(string) == 1:
        return string

    recursive_perms = []
    for c in string:
        for perm in permutations(string.replace(c,'',1)):
            recursive_perms.append(c+perm)

    return set(recursive_perms)


a = permutations("fookis")
print(a)