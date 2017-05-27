module type Queue = sig
  type 'a queue = 'a list * 'a list
 
  val empty : 'a queue
  val is_empty : 'a queue -> bool
 
  val snoc : 'a -> 'a queue -> 'a queue
  val first : 'a queue -> 'a option
  val rest : 'a queue -> 'a queue option
end
 
module TLQueue : Queue
