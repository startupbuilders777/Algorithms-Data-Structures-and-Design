module type OrderedType = sig
  type t
  val compare : t -> t -> int
end

module OrderedInt : (OrderedType with type t = int) =
  struct 
    type t = int
    let compare x y = if(x < y) then -1 else if (x = y) then 0 else 1
  end

module type mPQ = sig
  module Elem : OrderedType
 
  type 'a bintree =
    Empty
  | Node of int * 'a * 'a bintree * 'a bintree
 
  type mpq = Elem.t bintree
 
  (* type mpq *)
 
  val empty : mpq
  val isEmpty : mpq -> bool
  val merge : mpq -> mpq -> mpq
  val insert : Elem.t -> mpq -> mpq
  val findMin : mpq -> Elem.t option
  val deleteMin : mpq -> mpq option
end
 

module MaxiHeapmPQ (E : OrderedType) : (mPQ with module Elem = E) =
struct
  module Elem = E

   type 'a bintree =
    Empty
  | Node of int * 'a * 'a bintree * 'a bintree
 
  type mpq = Elem.t bintree
 
  let empty : mpq = Empty;; 
  
  let isEmpty (mhpq: mpq) : bool =
      match mhpq with
      | Empty -> true
      | _ -> false;;

    let rec merge (leftMPQ: mpq) (rightMPQ: mpq): mpq = 
        let mergeBranches root branchA branchB branchC =
            match (branchA, branchB, branchC) with
            | (Empty, Empty, Empty) -> Node(1, root, Empty, Empty)
            | (Empty, Empty, Node(s,_,_,_)) -> Node(1+s, root, Empty, branchC)
            | (Empty, Node(s,_,_,_), Empty) -> Node(1+s, root, Empty, branchB)
            | (Node(s,_,_,_), Empty, Empty) -> Node(1+s, root, Empty, branchA)
            | (Empty, Node(x,_,_,_), Node(y, _, _, _)) -> Node(1+x+y, root, branchB, branchC)
            | (Node(x,_,_,_), Empty, Node(y, _, _, _)) -> Node(1+x+y, root, branchA, branchC)
            | (Node(x,_,_,_), Node(y, _, _, _), Empty) -> Node(1+x+y, root, branchA, branchB)
            | (Node(aSize,_,_,_), Node(bSize,_,_,_), Node(cSize,_,_,_)) -> 
                begin
                    if(aSize >= bSize && aSize >= cSize) then Node(1+ aSize+bSize+cSize, root, branchA, merge(branchB)(branchC))
                    else if(bSize >= aSize && bSize >= cSize) then Node(1+ aSize+bSize+cSize, root, branchB, merge(branchA)(branchC))
                    else Node(1+ aSize+bSize+cSize, root, branchC, merge(branchA)(branchB))
                end in
        match (leftMPQ, rightMPQ) with
        | (Empty, Empty) -> Empty
        | (Empty, _) -> rightMPQ
        | (_, Empty) -> leftMPQ
        | (Node(sizeL, lval, ll, lr), Node(sizeR, rval, rl, rr)) -> if lval <= rval then mergeBranches(lval)(ll)(lr)(rightMPQ)    
                                                                    else  mergeBranches(rval)(rl)(rr)(leftMPQ)

let rec insert (element: Elem.t) (mhpq: mpq) : mpq =
    let elementTree = Node(1, element, Empty, Empty) in
    match mhpq with
    | Empty -> elementTree
    | Node (_, _, _, _) -> merge(elementTree)(mhpq);;

let findMin (mhpq: mpq) : Elem.t option =
    match mhpq with 
    | Node(_, v, _, _) -> Some v
    | Empty -> None;;


let deleteMin (mhpq: mpq) : mpq option =
  match mhpq with
  | Empty -> None
  | Node(_, v, left, right) -> Some(merge(left)(right)) ;;

end