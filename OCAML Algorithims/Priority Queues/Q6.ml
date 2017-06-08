
module type OrderedType = sig
  type t
  val compare : t -> t -> int
end

module OrderedInt: OrderedType = 
  struct 
    type t = int
    let compare x y = if(x < y) then -1 else if (x = y) then 0 else 1
  end

module type PQ = sig
  module Elem : OrderedType
 
  type 'a brt =
    Empty
  | Node of 'a * 'a brt * 'a brt
 
  type pq = Elem.t brt
 
  (* type pq *)
 
  val empty : pq
  val isEmpty : pq -> bool
 
  val insert : Elem.t -> pq -> pq
  val findMin : pq -> Elem.t option
  val deleteMin : pq -> pq option
end

module BHeapPQ (E : OrderedType) : (PQ with module Elem = E) = 
struct
  module Elem = E
  type 'a brt =
        Empty
        | Node of 'a * 'a brt * 'a brt
    
  type pq = Elem.t brt;; 
  
  let empty : pq = Empty;; 
  
  let isEmpty (bhpq: pq) : bool =
      match bhpq with
      | Empty -> true
      | _ -> false;;

  let rec insert (element: Elem.t) (bhpq: pq) : pq =
      match bhpq with
      | Empty -> Node(element, Empty, Empty) 
      | Node (v, left, right) -> if (Elem.compare(element)(v)) = -1 then Node(element, insert(v)(right), left) 
                              else Node(v, insert(element)(right), left) ;;

  let findMin (bhpq: pq) : Elem.t option =
      match bhpq with 
      | Node(v, left, right) -> Some v
      | Empty -> None;;

 (* let rec deleteOne (bhpq: pq) (newTree) = 
    let getFirstElementFromPair pair =
      match pair with
      | (first, second) -> second in
    let getSecondElementFromPair pair = 
      match pair with
      | (first, second) -> second in

    | Node(v, Empty, Empty) -> Some (v, newTree)
    | Node(v, left, right) ->  let result = deleteOne(left) in (Node(v, right, getSecondElementFromPair(left))) 
    | Empty -> None ;; 
*)
  let deleteMin (bhpq: pq) : pq option = Some bhpq ;;
end

(*
How do you use functor?


module IntPQ = BHeapPQ(OrderedInt);;

let i : OrderedInt.t = 2;;

 (*let braunHeapPQ = IntPQ.insert(2)(IntPQ.Empty) ;;*)
 *)