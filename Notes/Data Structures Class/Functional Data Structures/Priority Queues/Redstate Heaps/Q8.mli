module type OrderedType = sig
  type t
  val compare : t -> t -> int
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
 
module RsHeapmPQ (E : OrderedType) : (mPQ with module Elem = E)
