type 'a brt =
  Empty
| Node of 'a * 'a brt * 'a brt
 
type 'a sequence = 'a brt

val extend : 'a -> 'a sequence -> 'a sequence
val first : 'a sequence -> 'a option
val rest : 'a sequence -> 'a sequence option
val index : int -> 'a sequence -> 'a option

  type 'a sequence = 'a mylist
 
  let empty = Empty
 
  let isEmpty lst =
    match lst with  
    | Empty -> true
    | _ -> false
 
  let extend = fun e l -> Cons (e, l)
 
  let first = function
    | Empty -> None
    | Cons (e, l) -> Some e
 
  let rest = function
    | Empty -> None
    | Cons (e, l) -> Some l
 
  let rec index n brt = 
    match brt with
    | Empty -> None
    | Node (value, left, right) -> 
      match index with
      | 1 -> value
      | 

    when index = 0 -> value
    | No
    | Cons (e, l) when n = 0 -> Some e
    | Cons (e, Cons (_, tl)) when n > 0 -> index (n-1) tl
    | _ -> None
  
  let rec getBinaryBijective n =
    match n mod 2 with
    | 0 -> n mod 2 + 1
    | _ -> (getBinaryBijective(n/2));;


