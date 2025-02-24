import { ReactNode } from "react";

//type rafce press enter
interface Props {
  children: ReactNode; //Allow us to pass HtML content as an argument
}
const Alert = ({ children }: Props) => {
  return <div>{children}</div>;
};

export default Alert;
