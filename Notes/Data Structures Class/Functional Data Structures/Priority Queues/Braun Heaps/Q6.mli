
module type OrderedType = sig
  type t
  val compare : t -> t -> int
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
 
module BHeapPQ (E : OrderedType) : (PQ with module Elem = E)
