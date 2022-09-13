namespace CrossNolClient
{
	public class Game
	{
		public delegate void ProcNextStep();

		private int[,] gamefield;

		public static int SIZE = 3;

		private int playern;

		private ProcNextStep nextstep;

		public Game(int playern, ProcNextStep nextstep)
		{
			gamefield = new int[SIZE, SIZE];
			for (int i = 0; i < SIZE; i++)
			{
				for (int j = 0; j < SIZE; j++)
				{
					gamefield[i, j] = 0;
				}
			}
			this.playern = playern;
			this.nextstep = nextstep;
		}

		public string getPlayerFigure()
		{
			if (playern != 1)
			{
				return "нолики";
			}
			return "крестики";
		}

		public int getPlayerN()
		{
			return playern;
		}

		public int getField(int i, int j)
		{
			return gamefield[i, j];
		}

		public bool doClick(int i, int j, ref string command)
		{
			if (gamefield[i, j] == 0)
			{
				gamefield[i, j] = playern;
				command = $"click,{i},{j},{playern}";
				nextstep();
				return true;
			}
			return false;
		}

		public void setGameData(string data)
		{
			bool flag = false;
			int num = 0;
			for (int i = 0; i < SIZE; i++)
			{
				for (int j = 0; j < SIZE; j++)
				{
					if (gamefield[i, j] != int.Parse(data[num].ToString()))
					{
						gamefield[i, j] = int.Parse(data[num].ToString());
						flag = true;
					}
					num++;
				}
			}
			if (flag)
			{
				nextstep();
			}
		}

		public bool isGameOver(bool isvictory)
		{
			for (int i = 1; i <= 2; i++)
			{
				int num;
				for (int j = 0; j < SIZE; j++)
				{
					num = 0;
					for (int k = 0; k < SIZE; k++)
					{
						if (gamefield[j, k] == i)
						{
							num++;
						}
					}
					if (num == SIZE)
					{
						isvictory = playern == i;
						return true;
					}
					num = 0;
					for (int l = 0; l < SIZE; l++)
					{
						if (gamefield[l, j] == i)
						{
							num++;
						}
					}
					if (num == SIZE)
					{
						isvictory = playern == i;
						return true;
					}
				}
				num = 0;
				for (int m = 0; m < SIZE; m++)
				{
					if (gamefield[m, m] == i)
					{
						num++;
					}
				}
				if (num == SIZE)
				{
					isvictory = playern == i;
					return true;
				}
				num = 0;
				for (int n = 0; n < SIZE; n++)
				{
					if (gamefield[n, SIZE - 1 - n] == i)
					{
						num++;
					}
				}
				if (num == SIZE)
				{
					isvictory = playern == i;
					return true;
				}
			}
			return false;
		}
	}
}
