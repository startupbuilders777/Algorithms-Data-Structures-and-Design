type 'a brt =
  Empty
| Node of 'a * 'a brt * 'a brt

type 'a sequence = 'a brt

let empty = Empty;;
 
let isEmpty lst =
  match lst with  
  | Empty -> true
  | _ -> false;;
 
let rec extend newElement brt =
  match brt with 
  | Empty -> Node(newElement, Empty, Empty)
  | Node(element, left, right) 
      -> Node(newElement, (extend element right), left);;

let first brt =
  match brt with
  | Empty -> None
  | Node (value, left, right) -> Some value;;

let rec rest brt = 
    let rec aux brt = 
    match brt with
    | Empty -> Empty
    | Node (value, left, right) -> 
      match left with
      | Empty -> Empty
      | Node (lvalue, lleft, lright) -> Node(lvalue, right, (aux left)) in
  match brt with
  | Empty -> None
  | _ -> Some (aux brt)
 
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


let seq1 = Node (1, 
              Node(3, 
                Node(9, 
                  Node(3, Empty, Empty), 
                  Node(12, Empty, Empty)
                     ), 
                Node (10, 
                  Node(4, Empty, Empty), 
                  Node(15, Empty, Empty)
                     )
                  ), 
              Node(8, 
                Node(12, 
                  Node(19, Empty, Empty), 
                  Node(12, Empty, Empty)
                    ), 
                Node(3, 
                  Node(6, Empty, Empty), 
                  Node(1, Empty, Empty)
                    )
                )
           );;