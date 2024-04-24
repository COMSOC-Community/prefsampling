@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

REM You can set these variables from the command line, and also
REM from the environment for the first two.
SET SPHINXOPTS=
SET SPHINXBUILD=sphinx-build
SET SOURCEDIR=source
SET LOCALBUILDDIR=build
SET GITHUBBUILDDIR=..\docs

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

REM Put it first so that "make" without argument is like "make help".
:help
    %SPHINXBUILD% -M help "%SOURCEDIR%" "%LOCALBUILDDIR%" %SPHINXOPTS% %O%

:githubc
    echo. > "%GITHUBBUILDDIR%\.nojekyll"
    %SPHINXBUILD% -b html "%SOURCEDIR%" "%GITHUBBUILDDIR%" %SPHINXOPTS% %O%
:githubclean
    %SPHINXBUILD% -M clean "%SOURCEDIR%" "%GITHUBBUILDDIR%" %SPHINXOPTS% %O%

:html
    make doctest
    %SPHINXBUILD% -b html "%SOURCEDIR%" "%LOCALBUILDDIR%" %SPHINXOPTS% %O%

REM Catch-all target: route all unknown targets to Sphinx using the new
REM "make mode" option.  %O% is meant as a shortcut for %SPHINXOPTS%.
:all
    %SPHINXBUILD% -M %1 "%SOURCEDIR%" "%LOCALBUILDDIR%" %SPHINXOPTS% %O%

