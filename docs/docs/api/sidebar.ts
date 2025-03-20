import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebar: SidebarsConfig = {
  apisidebar: [
    {
      type: "doc",
      id: "api/ra-aid-api",
    },
    {
      type: "category",
      label: "sessions",
      items: [
        {
          type: "doc",
          id: "api/list-sessions-v-1-session-get",
          label: "List sessions",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/create-session-v-1-session-post",
          label: "Create session",
          className: "api-method post",
        },
        {
          type: "doc",
          id: "api/get-session-v-1-session-session-id-get",
          label: "Get session",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/get-session-trajectories-v-1-session-session-id-trajectory-get",
          label: "Get session trajectories",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "agent",
      items: [
        {
          type: "doc",
          id: "api/spawn-agent-v-1-spawn-agent-post",
          label: "Spawn agent",
          className: "api-method post",
        },
      ],
    },
    {
      type: "category",
      label: "UNTAGGED",
      items: [
        {
          type: "doc",
          id: "api/get-root-get",
          label: "Get Root",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "api/get-config-config-get",
          label: "Get Config",
          className: "api-method get",
        },
      ],
    },
  ],
};

export default sidebar.apisidebar;
