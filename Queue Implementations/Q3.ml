module type Queue = sig
  type 'a queue = 'a list * 'a list
 
  val empty : 'a queue
  val isEmpty : 'a queue -> bool
 
  val snoc : 'a -> 'a queue -> 'a queue
  val first : 'a queue -> 'a option
  val rest : 'a queue -> 'a queue option
end

module TLQueue : Queue = struct

type 'a queue = 'a list * 'a list

let empty = ([], []);;

let isEmpty queue =
    match queue with 
    | empty -> true
    | _ -> false;;

let first queue =
    match queue with
    | ([], []) -> None
    | (hd::tl, _) -> Some hd;;

let rec rest queue =
    let rec reverseList seq acc = 
        match seq with
        | [] -> acc
        | hd :: tl -> hd::acc in
    match queue with
    | ([], [])  -> None
    | ([], back) -> rest ((reverseList back []), [])
    | (hd::tl, back) -> Some (tl, back);;

let rec snoc element queue =
    let rec reverseList seq acc = 
        match seq with
        | [] -> acc
        | hd :: tl -> hd::acc in
    match queue with
    | ([], [])  -> ([element], [])
    | ([], back) -> snoc element ((reverseList back []), [])
    | (front, back) -> (front, element::back) ;;

end

let queue1 = (["a"; "b"; "c"], ["f", "e", "d"]);;

