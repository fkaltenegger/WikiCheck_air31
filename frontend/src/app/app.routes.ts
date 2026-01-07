import { Routes } from '@angular/router';
import { Home } from './routes/home/home';
import { FactCheck } from './routes/fact-check/fact-check';
import { History } from './routes/history/history';
import { Evaluation } from './routes/evaluation/evaluation';
import { HisoryResults } from './routes/hisory-results/hisory-results';

export const routes: Routes = [
    {
    path: '',
    component: Home
  },
 {
    path: 'fact-check',
    component: FactCheck
  },
  {
    path: 'history',
    component: History
  },
  {
    path: 'history-results/:id',
    component: HisoryResults
  },
  {
    path: 'evaluation',
    component: Evaluation
  },
  {
    path: '**',
    redirectTo: 'home',
  },
];
