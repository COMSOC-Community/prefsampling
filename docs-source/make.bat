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

if "%1" == "" goto help
if "%1" == "help" goto help
if "%1" == "github" goto github
if "%1" == "githubclean" goto githubclean
if "%1" == "html" goto html
goto all

:help
    %SPHINXBUILD% -M help "%SOURCEDIR%" "%LOCALBUILDDIR%" %SPHINXOPTS% %O%
    goto end

:github
    %SPHINXBUILD% -M doctest "%SOURCEDIR%" "%LOCALBUILDDIR%" %SPHINXOPTS% %O%
    echo. > "%GITHUBBUILDDIR%\.nojekyll"
    %SPHINXBUILD% -b html "%SOURCEDIR%" "%GITHUBBUILDDIR%" %SPHINXOPTS% %O%
    goto end

:githubclean
    %SPHINXBUILD% -M clean "%SOURCEDIR%" "%GITHUBBUILDDIR%" %SPHINXOPTS% %O%
    goto end

:html
    %SPHINXBUILD% -M doctest "%SOURCEDIR%" "%LOCALBUILDDIR%" %SPHINXOPTS% %O%
    %SPHINXBUILD% -b html "%SOURCEDIR%" "%LOCALBUILDDIR%" %SPHINXOPTS% %O%
    goto end

REM Catch-all target: route all unknown targets to Sphinx using the new
REM "make mode" option.  %O% is meant as a shortcut for %SPHINXOPTS%.
:all
    %SPHINXBUILD% -M %1 "%SOURCEDIR%" "%LOCALBUILDDIR%" %SPHINXOPTS% %O%
    goto end

:end

popd

