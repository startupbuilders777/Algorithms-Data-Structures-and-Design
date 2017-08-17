'''
Write a method to find the frequency of occurences of any given word in a book.
What if we were running this algo multiple times? -> Do some preprocessing to make it faster O(1) access for any word
'''

book = ( "VirtualBox imposes no limits on the number of snapshots you can take. The only practical " ) + (
    ( "limitation is disk space on your host: each snapshot stores the state of the virtual machine" ) + (
        ("and thus occupies some disk space. (See the next section for details on what exactly is" ) + (
            ("stored in a snapshot.)" ) + (
                ("of snapshots. By restoring a snapshot, you go back (or forward) in time: the current state" ) + (
                    ("2. You can restore a snapshot by right-clicking on any snapshot you have taken in the list" ) + (
                        ("snapshot was taken.4"  ) + (
                            ("Note: Restoring a snapshot will affect the virtual hard drives that are connected to your" ) + (
                                ("of the machine is lost, and the machine is restored to the exact state it was in when the" ) + (
                                    ("that all files that have been created since the snapshot and all other file changes will be" ) + (
                                        ("VM, as the entire state of the virtual hard drive will be reverted as well. This means also" ) + (
                                            ("lost. In order to prevent such data loss while still making use of the snapshot feature, it" ) + (
                                                ("is possible to add a second hard drive in write-through mode using the VBoxManage" ) + (
                                                    ("interface and use it to store your data. As write-through hard drives are not included in" ) + (
                                                        ("snapshots, they remain unaltered when a machine is reverted. See chapter 5.4, Special" ) + (
                                                            ("image write modes, page 85 for details." ) + (
                                                                ("To avoid losing the current state when restoring a snapshot, you can create a new snapshot" ) + (
                                                                    "before the restore." ) + ("By restoring an earlier snapshot and taking m" )))))))))))))))))

class parser():
    def __init__(self, word):
        self.word = word
        self.frequency = self.count(word)

    def count(self, word):
        freq = {}
        word = word.split()
        for i in word:
            freq[i] = 0

        for i in word:
            freq[i] += 1
        return freq

    def wordCount(self, word):
        return self.frequency.get(word)

wordParser = parser(word=book)
print(wordParser.wordCount("the"))
print(wordParser.frequency)