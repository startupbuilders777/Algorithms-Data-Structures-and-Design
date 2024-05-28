import sys

def filter_git_diff(diff_input):
    """
    Filter a git diff to keep only the additions.
    
    :param diff_input: An iterable with lines of the git diff
    :return: List of lines that are additions
    """
    additions = []
    for line in diff_input:
        if line.startswith('+') and not line.startswith('+++'):
            additions.append(line)
    return additions

if __name__ == "__main__":
    # Read from standard input if no file is provided
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            diff_input = file.readlines()
    else:
        diff_input = sys.stdin.readlines()
    
    additions = filter_git_diff(diff_input)
    
    for line in additions:
        print(line, end='')
