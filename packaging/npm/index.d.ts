export interface SolveOptions {
  binary?: string; device?: 'cpu' | 'cuda' | 'auto'; method?: string;
  timeLimit?: number; tolerance?: number; threads?: number; signal?: AbortSignal;
}
export interface SolveResult {status: string; objective: number | null; [key: string]: unknown;}
export function solve(model: string | Record<string, unknown>, options?: SolveOptions): Promise<SolveResult>;
export function executable(binary?: string): string;

export function renderReport(result: SolveResult, options?: {title?: string}): string;
