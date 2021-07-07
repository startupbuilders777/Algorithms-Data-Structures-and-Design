type 'a brt =
  Empty
| Node of 'a * 'a brt * 'a brt
 
type 'a sequence = 'a brt
 
val empty : 'a sequence
val isEmpty : 'a sequence -> bool
val extend : 'a -> 'a sequence -> 'a sequence
val first : 'a sequence -> 'a option
val rest : 'a sequence -> 'a sequence option
val index : int -> 'a sequence -> 'a option
