
module type OrderedType = sig
  type t
  val compare : t -> t -> int
end

module OrderedInt : (OrderedType with type t = int) =
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

  let rec deleteOne (bhpq) = 
    let getFirstElementFromPair pair =
      match pair with
      | (first, second) -> first in
    let getSecondElementFromPair pair = 
      match pair with
      | (first, second) -> second in
    match bhpq with
    | Node(v, Empty, Empty) -> (v, Empty)
    | Node(v, left, right) ->  let result = deleteOne(left) in
                               let element = getFirstElementFromPair(result) in
                               let tree = getSecondElementFromPair(result) in
                               (element, (Node(v, right, tree))) ;;

let rec meld element leftTree rightTree = 
  match (leftTree, element, rightTree) with
  | (Empty, element, Empty) -> Node(element, empty, empty)
  | (Node(lval, ll, lr), element, Empty) -> if(element <= lval) then Node(element, meld(lval)(ll)(Empty), lr)
                                            else Node(lval, meld(element)(ll)(Empty), lr)

  | (Empty, element, Node(rval, rl, rr)) -> if(element <= rval) then Node(element, rl, meld(rval)(Empty)(rr))
                                            else Node(rval, rl, meld(element)(rr)(Empty))
  | (Node(lval, ll, lr), element, Node(rval, rl, rr)) -> 
    begin
      if (element <= rval && element <= lval) then Node(element, leftTree, rightTree)
      else if (rval <= element && rval <= lval) then Node(rval, leftTree, meld(element)(rl)(rr))
      else Node(lval, meld(element)(ll)(lr), rightTree)
  end;;

let deleteMin (bhpq: pq) : pq option =
  match bhpq with
  | Empty -> None
  | _ -> 
    begin
    match deleteOne(bhpq) with
      | (v, Empty) -> Some Empty
      | (displacedElement, Node(min, left, right)) -> Some(meld(displacedElement)(left)(right))
    end;;
    
end

module IPQ = BHeapPQ(OrderedInt)

let p1 = IPQ.Node(1, Empty, Empty);;
let p2 = IPQ.Node(1, IPQ.Node(2, IPQ.Empty, IPQ.Empty), IPQ.Node(3, IPQ.Empty, IPQ.Empty));;
let p3 = IPQ.Node(2, IPQ.Node(3, IPQ.Empty, IPQ.Empty), Empty);;

let p4 = IPQ.Node(1, IPQ.Node(2, IPQ.Node(6, IPQ.Empty, IPQ.Empty), IPQ.Node(7, IPQ.Empty, IPQ.Empty)), 
                     IPQ.Node(3, IPQ.Node(8, IPQ.Empty, IPQ.Empty), IPQ.Node(9, IPQ.Empty, IPQ.Empty)));;

let p5 =  (IPQ.Node (2, IPQ.Node (3, IPQ.Node (8, IPQ.Empty, IPQ.Empty), IPQ.Node (9, IPQ.Empty, IPQ.Empty)),                                
                        IPQ.Node (6, IPQ.Node (7, IPQ.Empty, IPQ.Empty), IPQ.Empty)))  

let () = 
  assert(IPQ.deleteMin(p2) = Some p3)
let () =
  assert(IPQ.deleteMin(p4) = Some p5)