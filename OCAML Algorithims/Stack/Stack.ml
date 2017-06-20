module type Stack = sig
 type 'a stack =
    Empty
  | Cons of 'a * 'a stack ;;

  val empty : 'a stack
  val is_empty : 'a stack -> bool

  val push : 'a -> 'a stack -> 'a stack
  val front : 'a stack -> 'a option
  val pop : 'a stack -> 'a stack option
end

module MyStack : Stack = struct

   type 'a stack =
    Empty
  | Cons of 'a * 'a stack ;;

let empty = Empty;;
let is_empty stack =  
  match stack with 
  | Empty -> true
  | _ -> false ;;

let push element stack = 
  match stack with 
  | Empty -> Cons(element, Empty)
  | _ -> Cons(element, stack);;
  
let pop stack = 
  match stack with
  | Empty -> None
  | Cons(a, restOfStack) -> Some restOfStack ;;

let front stack =
  match stack with 
  | Empty -> None
  | Cons(front, restOfStack) -> Some front ;;
end

