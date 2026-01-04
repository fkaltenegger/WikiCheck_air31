import { Component } from '@angular/core';
import { Input } from '../../components/input/input';
import { Results } from '../../components/results/results';

@Component({
  selector: 'app-fact-check',
  imports: [Input, Results],
  templateUrl: './fact-check.html',
  styleUrl: './fact-check.css',
})
export class FactCheck {

}
