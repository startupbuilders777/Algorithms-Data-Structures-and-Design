type 'a brt =
  Empty
| Node of 'a * 'a brt * 'a brt

let empty = Empty;;
 
let isEmpty lst =
  match lst with  
  | Empty -> true
  | _ -> false;;
 
let rec extend brt newElement =
  match brt with 
  | Empty -> Node(newElement, Empty, Empty)
  | Node(element, left, right) 
      -> Node(newElement, (extend right element), left);;

let first brt =
  match brt with
  | Empty -> None
  | Node (value, left, right) -> Some value;;
 
let rec rest brt = 
  match brt with
  | Empty -> Empty
  | Node (value, left, right) -> 
    match left with
    | Empty -> Empty
    | Node (lvalue, lleft, lright) -> Node(lvalue, right, (rest left));;
 
let rec index n brt = 
  match brt with
  | Empty -> None
  | Node (value, left, right) -> 
    if n <= 0 
      then Some value
    else 
      match n mod 2 = 0 with
      | true -> index((n-2)/2)(right)
      | false -> index((n-1)/2)(left);;   

let seq1 = 
  Node(0,   
    Node(1, 
      Node(3, Empty, Empty), 
      Node(5, Empty, Empty)), 
    Node(2,  
      Node(4, Empty, Empty), 
      Node(6, Empty, Empty))
      );;

let() = assert(index 0 seq1 = Some 0);;
let() = assert(index 1 seq1 = Some 1);;
let() = assert(index 2 seq1 = Some 2);;
let() = assert(index 3 seq1 = Some 3);;
let() = assert(index 4 seq1 = Some 4);;
let() = assert(index 5 seq1 = Some 5);;
let() = assert(index 6 seq1 = Some 6);;

let extendedSeq1 = extend seq1 9;;
let() = assert(index 0 extendedSeq1 = Some 9);;
let() = assert(index 1 extendedSeq1 = Some 0);;
let() = assert(index 2 extendedSeq1 = Some 1);;
let() = assert(index 3 extendedSeq1 = Some 2);;
let() = assert(index 4 extendedSeq1 = Some 3);;
let() = assert(index 5 extendedSeq1 = Some 4);;
let() = assert(index 6 extendedSeq1 = Some 5);;
let() = assert(index 7 extendedSeq1 = Some 6);;

let seq2 = rest extendedSeq1

let() = assert(index 0 seq2 = Some 0);;
let() = assert(index 1 seq2 = Some 1);;
let() = assert(index 2 seq2 = Some 2);;
let() = assert(index 3 seq2 = Some 3);;
let() = assert(index 4 seq2 = Some 4);;
let() = assert(index 5 seq2 = Some 5);;
let() = assert(index 6 seq2 = Some 6);;
