import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  route("/", "routes/pageLayout.tsx", [
    index("routes/pages/home.tsx"),
    route("rdlnn", "routes/pages/rdlnn/rdlnn-main-page.tsx"),
    route("dwt", "routes/pages/dwt/dwt.tsx"),
    route("dywt", "routes/pages/dywt/dywt.tsx"),
  ]),
] satisfies RouteConfig;
