import React from "react";
import ListGroup from "./components/ListGroup";
import Alert from "./components/Alert";
import Button from "./components/button";
function App() {
  let items = ["New York", "San Francisco", "Tokyo", "London", "Paris"];
  //Call ListGroup and input arguments
  const handleSelectItem = (item: String) => {
    console.log(item);
  };
  return (
    <div>
      <ListGroup
        items={items}
        heading="Cities"
        onSelectItem={handleSelectItem}
      />
      <div className="alert alert-primary">
        <Alert>
          Hello <span>World</span>
        </Alert>
      </div>
      <Button onClick={() => console.log("Clicked")}>My Button</Button>
    </div>
  );
}

export default App;
