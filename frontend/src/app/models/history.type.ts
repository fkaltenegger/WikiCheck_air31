import { Query } from "./query.type";
import { ResultItem } from "./result.type";

export interface HistoryItem {
    query: Query;
    results: ResultItem[];
}