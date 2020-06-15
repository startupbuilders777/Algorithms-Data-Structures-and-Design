module type Queue = sig
  type 'a queue = 'a list * 'a list

  val empty : 'a queue
  val is_empty : 'a queue -> bool

  val snoc : 'a -> 'a queue -> 'a queue
  val first : 'a queue -> 'a option
  val rest : 'a queue -> 'a queue option
end

module TLQueue : Queue = struct

  type 'a queue = 'a list * 'a list

  let empty = ([], []);;

  let is_empty queue =
    match queue with 
    | ([],[]) -> true
    | _ -> false;;

  let first queue =
    match queue with
    | ([], _) -> None
    | (hd::tl, _) -> Some hd;;

  let rec rest queue =
    let rec reverseList seq acc = 
      match seq with
      | [] -> acc
      | hd :: tl -> reverseList(tl)(hd::acc) in
    match queue with
    | ([], _)  -> None
    | ([], back) -> rest ((reverseList back []), [])
    | (hd::tl, back) -> if tl = [] then Some ((reverseList back []), [])
                        else Some (tl, back);;

  let rec snoc element queue =
      let rec reverseList seq acc = 
      match seq with
      | [] -> acc
      | hd :: tl -> reverseList(tl)(hd::acc) in
    match queue with
    | ([], [])  -> ([element], [])
    | ([], back) -> ((reverseList (element::back) []), [])
    | (front, back) -> (front, element::back) ;;
end

let queue1 = (["a"; "b"; "c"], ["f"; "e"; "d"]);;

let get value =
  match value with 
  | Some x -> x;;

let() = assert( TLQueue.first queue1 = Some "a");;

let() = assert( TLQueue.snoc "a" ([], []) = (["a"], []));;
let() = assert( TLQueue.first queue1 = Some "a");;
let() = assert( TLQueue.first queue1 = Some "a");;

let() = assert( TLQueue.rest queue1 = Some (["b"; "c"], ["f"; "e"; "d"]));;
let() = assert( (TLQueue.rest (get (TLQueue.rest queue1)))  = Some (["c"], ["f"; "e"; "d"])) ;;


let() = assert( 
 (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest queue1)))))))= Some(["e"; "f"], [])
)

let() = assert( 
  (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest queue1)))))))))= Some(["f"], [])
)
let() = assert( 
  (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest queue1)))))))))))= Some([], [])
)
let() = assert( 
  (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest (get (TLQueue.rest queue1)))))))))))))= None
)



