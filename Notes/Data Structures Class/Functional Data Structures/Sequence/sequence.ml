#use "sequence.mli"

module MySeq : Sequence = struct
 
  type 'a mylist =
    Empty
  | Cons of 'a * 'a mylist
 
  type 'a sequence = 'a mylist
 
  let empty = Empty
 
  let isEmpty = function
    | Empty -> true
    | _ -> false
 
  let extend = fun e l -> Cons (e, l)
 
  let first = function
    | Empty -> None
    | Cons (e, l) -> Some e
 
  let rest = function
    | Empty -> None
    | Cons (e, l) -> Some l
 
  let rec index n = function
    | Empty -> None
    | Cons (e, l) when n = 0 -> Some e
    | Cons (e, Cons (_, tl)) when n > 0 -> index (n-1) tl
    | _ -> None
 
end
