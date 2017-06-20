type 'a stack =
    Empty
  | Cons of 'a * 'a stack ;;

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
  