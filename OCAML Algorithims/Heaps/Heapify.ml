type 'a bintree =
    Empty
  | Node of int * 'a * 'a bintree * 'a bintree

list[1;3;4;5;6;7;4;3;1;2;3];;

let meld element lh rh =
  let getSize heap =
    match heap with
    | Node(sz, _, _, _) -> sz
    | Empty -> 0 in
  let totalSize = 1 + getSize(lh) + getSize(rh) in
  match (lh, rh) with
  | (Node(lval, ll, lr), Node(rval, rl, rr)) -> 
    begin 
      if(element >= lval && element >= rval) then Node(totalSize, element, lh, rh)    
      else if(lval >= element && lval >= rval) then Node(totalSize, lval, )
