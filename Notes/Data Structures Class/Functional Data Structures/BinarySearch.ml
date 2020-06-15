
(*List has to be sorted for binarySearch to work
Cant binary search a list, only an array that you can index into quickly
...
*)

let rec binarySearch lst element =
    match lst with 
    | [] -> None
    