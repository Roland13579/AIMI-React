import React, { useState } from "react";
// {items: [], heading: str}
interface Props {
  items: string[];
  heading: string;
  //onselectitem, function of input type string and output void - ususally, when user press a button, smth will happen, function simulates this
  onSelectItem: (item: string) => void;
}
function ListGroup({ items, heading, onSelectItem }: Props) {
  //argument
  //In react, you cannot return more than 1 element, so you cannot add other element like h1 etc.
  //So you use <> </> to wrap all the elements under 1 element, react will treat it as 1 element
  //let items = ["New York", "San Francisco", "Tokyo", "London", "Paris"];
  //Hook - telling react the value or state changes overtime
  const [selectedIndex, setSelectedIndex] = useState(-1);

  //JSX looks like HTMl inside Javascript, but its not Javascript. IT does not support for & if loops, so you cannot use for loop inside JSX
  //You can use map function to loop through the array and return the JSX
  // item => <li>{item}</li> is a arrow function, which takes item as input and returns <li>{item}</li>
  // key={item} is used to uniquely identify the element, so that react can identify which element to update, delete etc.
  //For if statement, use {} and the condition syntaxes
  return (
    <>
      <h1>{heading}</h1>
      {items.length === 0 ? <p>No item found</p> : null}
      <ul className="list-group">
        {items.map((item, index) => (
          <li
            className={
              selectedIndex === index
                ? "list-group-item active"
                : "list-group-item"
            }
            key={item}
            onClick={() => {
              setSelectedIndex(index);
              onSelectItem(item);
            }}
          >
            {item}
          </li>
        ))}
      </ul>
    </>
  );
}

export default ListGroup;
