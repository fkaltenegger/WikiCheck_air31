import { Routes } from '@angular/router';
import { Home } from './routes/home/home';
import { FactCheck } from './routes/fact-check/fact-check';
import { History } from './routes/history/history';

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
    path: '**',
    redirectTo: 'home',
  },
];
