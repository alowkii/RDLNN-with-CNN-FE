import { NavLink, Link } from "react-router";

export default function Header() {
  return (
    <header className="flex items-center justify-between bg-silversand p-4 text-gunmetal sticky w-full">
      <h1 className="text-2xl font-bold">
        <NavLink
          to={"rdlnn"}
          className={({ isActive }) => (isActive ? "active" : "")}
        >
          RDLNN
        </NavLink>{" "}
        <span className="text-metallicsilver">vs</span>{" "}
        <NavLink
          to={"dwt"}
          className={({ isActive }) => (isActive ? "active" : "")}
        >
          DWT
        </NavLink>{" "}
        <span className="text-metallicsilver">vs</span>{" "}
        <NavLink
          to={"dywt"}
          className={({ isActive }) => (isActive ? "active" : "")}
        >
          DyWT
        </NavLink>{" "}
      </h1>
      <nav>
        <Link className="text-gunmetal" to={"/"}>
          HOME
        </Link>
      </nav>
    </header>
  );
}
