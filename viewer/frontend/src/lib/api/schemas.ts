import { z } from "zod";

export const configValueSchema = z.union([z.string(), z.number(), z.boolean(), z.null()]);

export type ConfigValue = z.infer<typeof configValueSchema>;
